import SwiftUI

enum AIDBTokens {
    enum Radius {
        static let small: CGFloat = 8
        static let medium: CGFloat = 12
        static let large: CGFloat = 18
    }

    enum Padding {
        static let xSmall: CGFloat = 6
        static let small: CGFloat = 12
        static let medium: CGFloat = 16
        static let large: CGFloat = 24
    }

    enum Shadow {
        static let card = Color.black.opacity(0.16)
    }

    static let stroke = Color.white.opacity(0.18)
    static let separator = Color.white.opacity(0.08)
}

enum AIDBColors {
    static let accent = Color(red: 0.38, green: 0.55, blue: 0.98)
    static let accentGradient = LinearGradient(
        colors: [Color(red: 0.33, green: 0.51, blue: 0.96), Color(red: 0.58, green: 0.36, blue: 0.94)],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )
    static let success = Color(red: 0.33, green: 0.78, blue: 0.49)
    static let warning = Color(red: 0.97, green: 0.74, blue: 0.27)
    static let destructiveGradient = LinearGradient(
        colors: [Color(red: 0.91, green: 0.31, blue: 0.35), Color(red: 0.68, green: 0.21, blue: 0.34)],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )
    static let surface = Color(nsColor: .controlBackgroundColor)
    static let elevatedSurface = Color(red: 0.12, green: 0.14, blue: 0.19)
    static let shadow = Color.black.opacity(0.22)
}

struct PrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(
                EdgeInsets(
                    top: AIDBTokens.Padding.small,
                    leading: AIDBTokens.Padding.medium,
                    bottom: AIDBTokens.Padding.small,
                    trailing: AIDBTokens.Padding.medium
                )
            )
            .background(
                AIDBColors.accentGradient
                    .opacity(configuration.isPressed ? 0.8 : 1.0), in: Capsule()
            )
            .overlay(
                Capsule().stroke(Color.white.opacity(configuration.isPressed ? 0.2 : 0.12), lineWidth: 0.6)
            )
            .foregroundStyle(.white)
            .font(.system(.callout, weight: .semibold))
            .shadow(color: AIDBColors.shadow.opacity(configuration.isPressed ? 0.2 : 0.35), radius: configuration.isPressed ? 6 : 10, x: 0, y: configuration.isPressed ? 2 : 8)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
    }
}

struct SecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(
                EdgeInsets(
                    top: AIDBTokens.Padding.xSmall,
                    leading: AIDBTokens.Padding.medium,
                    bottom: AIDBTokens.Padding.xSmall,
                    trailing: AIDBTokens.Padding.medium
                )
            )
            .background(
                RoundedRectangle(cornerRadius: AIDBTokens.Radius.medium, style: .continuous)
                    .fill(Color.white.opacity(0.06))
                    .overlay(
                        RoundedRectangle(cornerRadius: AIDBTokens.Radius.medium, style: .continuous)
                            .stroke(Color.white.opacity(configuration.isPressed ? 0.26 : 0.18), lineWidth: 1)
                    )
            )
            .foregroundStyle(.primary)
            .font(.system(.callout, weight: .medium))
            .shadow(color: AIDBTokens.Shadow.card.opacity(configuration.isPressed ? 0.12 : 0.22), radius: configuration.isPressed ? 3 : 6, x: 0, y: configuration.isPressed ? 1 : 4)
            .scaleEffect(configuration.isPressed ? 0.985 : 1)
    }
}

struct GlassFieldStyle: ViewModifier {
    var isValid: Bool

    func body(content: Content) -> some View {
        content
            .textFieldStyle(.plain)
            .padding(.horizontal, AIDBTokens.Padding.small)
            .padding(.vertical, AIDBTokens.Padding.xSmall)
            .background(
                RoundedRectangle(cornerRadius: AIDBTokens.Radius.medium, style: .continuous)
                    .fill(Color.white.opacity(0.08))
            )
            .overlay(
                RoundedRectangle(cornerRadius: AIDBTokens.Radius.medium, style: .continuous)
                    .stroke(isValid ? AIDBColors.accent.opacity(0.35) : Color.white.opacity(0.18), lineWidth: 1)
            )
    }
}

extension View {
    func glassCard(cornerRadius: CGFloat = AIDBTokens.Radius.large, strokeColor: Color = AIDBTokens.stroke) -> some View {
        modifier(GlassCardModifier(cornerRadius: cornerRadius, strokeColor: strokeColor))
    }

    func glassField(isValid: Bool = true) -> some View {
        modifier(GlassFieldStyle(isValid: isValid))
    }
}

private struct GlassCardModifier: ViewModifier {
    var cornerRadius: CGFloat
    var strokeColor: Color

    func body(content: Content) -> some View {
        content
            .padding(AIDBTokens.Padding.medium)
            .background(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(Color.white.opacity(0.04))
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .stroke(strokeColor, lineWidth: 1)
            )
            .shadow(color: AIDBTokens.Shadow.card, radius: 14, x: 0, y: 10)
    }
}

struct StatusPill: View {
    enum Tone {
        case success
        case warning
        case neutral
    }

    let tone: Tone
    let label: String
    let systemImage: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: systemImage)
                .font(.system(size: 12, weight: .bold))
            Text(label)
                .font(.system(.caption, weight: .semibold))
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 12)
        .background(background, in: Capsule())
        .foregroundStyle(.white)
        .overlay(Capsule().stroke(Color.white.opacity(0.2), lineWidth: 0.6))
    }

    private var background: some ShapeStyle {
        switch tone {
        case .success:
            return LinearGradient(colors: [AIDBColors.success, AIDBColors.success.opacity(0.8)], startPoint: .top, endPoint: .bottom)
        case .warning:
            return LinearGradient(colors: [AIDBColors.warning, AIDBColors.warning.opacity(0.86)], startPoint: .top, endPoint: .bottom)
        case .neutral:
            return LinearGradient(colors: [Color.gray.opacity(0.7), Color.gray.opacity(0.5)], startPoint: .top, endPoint: .bottom)
        }
    }
}

struct AdaptiveBlurBackground: View {
    var body: some View {
        RoundedRectangle(cornerRadius: AIDBTokens.Radius.large, style: .continuous)
            .fill(Color.white.opacity(0.05))
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: AIDBTokens.Radius.large, style: .continuous))
    }
}
